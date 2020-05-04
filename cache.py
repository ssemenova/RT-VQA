import sqlite3
from chunk import Chunk

class Cache(object):
  def __init__(self, db_name, cache_size, evict_mod, use_ram):
    self.use_ram = use_ram
    self.cache_size = cache_size
    self.evict_mod = evict_mod

    self.oldest_id = 1
    self.newest_id = 0

    self.current_id = 1
    self.current_chunk = None

    if use_ram:
      self.db = {}
    else:
      self.db_name = db_name + '.db'
      self.conn = _connect()
      cursor = self.conn.cursor()    
      cursor.execute("create table videochunks(id, vgg, c3d)")
  
  def __getitem__(self, id):
    if self.use_ram:
      chunk = self.db.get(id)
      return chunk
    else:
      cursor = self.conn.cursor()
      t = (id,)
      cursor.execute('SELECT * FROM videochunks WHERE id=?', t)
      results = cursor.fetchone()
      # TODO: later, return something other than this so it matches the above
      return results[1], results[2]

  def _connect(self):
    return sqlite3.connect(db_name + '.db')

  def commit(self):
    self.current_chunk.generate_features()
    self.newest_id = self.current_chunk.id

  def new_chunk(
    self, chunk_size, frames_per_clip_c3d, 
    clip_num_c3d, i3d_extractor_model_path
  ):
    self.current_chunk = Chunk(
      self.current_id, chunk_size,
      frames_per_clip_c3d, clip_num_c3d,
      i3d_extractor_model_path,
    )

    self.db.update({
        self.current_chunk.id: self.current_chunk
    })

    self.current_id +=1

    # Loosely evict from the cache every [evict_mod] chunks
    oldest_allowed_id = max(
        1, self.current_chunk.id - self.cache_size
    )
    should_evict = oldest_allowed_id % self.evict_mod == 0
    if should_evict:
      self._evict(oldest_allowed_id)
      self.oldest_id = oldest_allowed_id

  def size(self):
    return str(self.oldest_id - self.newest_id)

  # TODO: get rid of this method
  def insert_latest_chunk(self):
    if self.use_ram:
      self.db.update({
          self.current_chunk.id: self.current_chunk
      })
    else:
      # TODO: later, is self a valid way to insert vgg and c3d data?
      cursor = self.conn.cursor()
      cursor.execute(
        "INSERT INTO videochunks VALUES(" +
        "'" + self.current_chunk.id + "', '" +
        self.current_chunk.vgg_features + "', '" +
        self.current_chunk.c3d_features + "'"
        + ")"
      )
      self.conn.commit()

    self.newest_id += 1

    # Loosely evict from the cache every [evict_mod] chunks
    oldest_allowed_id = max(
        1, self.current_chunk.id - self.cache_size
    )
    should_evict = oldest_allowed_id % self.evict_mod == 0
    if should_evict:
      self._evict(oldest_allowed_id)
      self.oldest_id = oldest_allowed_id

  def _evict(self, oldest_allowed_id):
    if self.use_ram:
      keep_evicting = True
      current_id = oldest_allowed_id - 1

      while keep_evicting:
        if not self.db.pop(current_id, None):
          keep_evicting = False
        else:
          current_id -= 1
    else:
      cursor = self.conn.cursor()
      t = (oldest_allowed_id,)
      cursor.execute('DELETE FROM videochunks WHERE id<', t)
      self.conn.commit()
